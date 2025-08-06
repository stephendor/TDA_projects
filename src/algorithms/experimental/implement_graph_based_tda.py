#!/usr/bin/env python3
"""
Graph-Based Network Topology TDA Implementation
Phase 2A of Advanced TDA Enhancement Strategy

This implements topological analysis of network connection graphs
to detect APT lateral movement and C&C communication patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import TDA modules
# Updated imports for new structure
from ...core.persistent_homology import PersistentHomologyAnalyzer

class NetworkGraphTDAAnalyzer:
    """
    Advanced TDA analyzer that applies topological analysis to
    network connection graphs rather than just flow features.
    """
    
    def __init__(self, graph_window_sizes=None, min_connections=2):
        """
        Initialize graph-based TDA analyzer.
        
        Args:
            graph_window_sizes: Time windows for graph construction
            min_connections: Minimum connections to include a node
        """
        if graph_window_sizes is None:
            # Different temporal scales for graph analysis
            self.graph_window_sizes = [
                20,   # Short-term: Immediate connections
                50,   # Medium-term: Local network patterns
                100,  # Long-term: Campaign-level topology
                200   # Strategic: Persistent infrastructure
            ]
        else:
            self.graph_window_sizes = graph_window_sizes
            
        self.min_connections = min_connections
        self.tda_analyzers = {}
        
        print(f"🕸️ Graph-Based Network TDA Analyzer initialized")
        print(f"   Graph window sizes: {self.graph_window_sizes}")
        print(f"   Minimum connections per node: {min_connections}")

    def extract_network_graphs(self, df):
        """Extract temporal sequence of network graphs from flow data."""
        
        print(f"\n📊 EXTRACTING NETWORK GRAPHS FROM FLOW DATA")
        print("=" * 60)
        
        # Sort by time (using index as proxy for time ordering)
        df_sorted = df.sort_index()
        
        all_graph_features = []
        all_labels = []
        
        for window_idx, window_size in enumerate(self.graph_window_sizes):
            print(f"\n   📏 Graph Scale {window_idx + 1}: Window size {window_size}")
            
            # Create graph sequences for this window size
            graph_sequences, graph_labels = self.create_graph_sequences(
                df_sorted, window_size
            )
            
            if len(graph_sequences) == 0:
                print(f"      ⚠️ No graphs generated for window size {window_size}")
                continue
                
            print(f"      Generated {len(graph_sequences)} network graphs")
            
            # Extract TDA features from graph sequences
            graph_tda_features = self.extract_graph_tda_features(
                graph_sequences, window_idx
            )
            
            if graph_tda_features is not None:
                all_graph_features.append(graph_tda_features)
                all_labels.append(graph_labels)
                print(f"      ✅ Graph TDA features: {graph_tda_features.shape}")
            else:
                print(f"      ❌ Graph TDA feature extraction failed")
        
        if not all_graph_features:
            print("\n❌ No graph features extracted at any scale")
            return None, None
        
        # Combine features from all graph scales
        # Use the scale with best attack preservation
        attack_rates = [np.mean(labels) for labels in all_labels]
        best_scale_idx = np.argmax(attack_rates)
        
        print(f"   Attack rates by graph scale: {[f'{rate:.3%}' for rate in attack_rates]}")
        print(f"   Using graph scale {best_scale_idx + 1} (window {self.graph_window_sizes[best_scale_idx]}) as primary")
        
        # Use primary scale and augment with other scales
        primary_features = all_graph_features[best_scale_idx]
        primary_labels = all_labels[best_scale_idx]
        
        # Add complementary features from other scales
        n_samples = len(primary_features)
        additional_features = []
        
        for scale_idx, features in enumerate(all_graph_features):
            if scale_idx != best_scale_idx and len(features) >= n_samples:
                additional_features.append(features[:n_samples])
        
        if additional_features:
            combined_features = np.concatenate([primary_features] + additional_features, axis=1)
            print(f"   Combined graph features from {len(additional_features) + 1} scales")
        else:
            combined_features = primary_features
            print(f"   Using only primary graph scale features")
        
        combined_labels = primary_labels
        
        print(f"\n📊 GRAPH TDA FEATURE SUMMARY:")
        print(f"   Final feature matrix: {combined_features.shape}")
        print(f"   Total attack sequences: {np.sum(combined_labels)}")
        print(f"   Total benign sequences: {np.sum(combined_labels == 0)}")
        print(f"   Attack rate: {np.mean(combined_labels):.3%}")
        
        return combined_features, combined_labels

    def create_graph_sequences(self, df, window_size, step_size=None):
        """Create sequence of network graphs from flow data."""
        
        if step_size is None:
            step_size = max(1, window_size // 4)  # Overlap for better coverage
        
        if len(df) < window_size:
            return [], []
        
        graph_sequences = []
        labels = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            # Extract window of flows
            window_flows = df.iloc[i:i+window_size]
            
            # Build network graph for this window
            G = self.build_network_graph(window_flows)
            
            if G.number_of_nodes() < 3:  # Need minimum nodes for TDA
                continue
                
            # Determine label for this graph
            window_labels = window_flows.get('Label', pd.Series(['BENIGN'] * len(window_flows)))
            if hasattr(window_labels, 'values'):
                label_values = window_labels.values
            else:
                label_values = window_labels
                
            # If ANY attack in window, label as attack
            is_attack = any(label != 'BENIGN' for label in label_values if label != 'BENIGN')
            sequence_label = 1 if is_attack else 0
            
            graph_sequences.append(G)
            labels.append(sequence_label)
        
        return graph_sequences, np.array(labels)

    def build_network_graph(self, flows_window):
        """Build weighted network graph from flows."""
        
        G = nx.Graph()
        
        # Build graph from network flows
        for _, flow in flows_window.iterrows():
            try:
                src_ip = str(flow.get('Source IP', f'src_{len(G.nodes)}'))
                dst_ip = str(flow.get('Destination IP', f'dst_{len(G.nodes)}'))
                
                # Calculate edge weight from multiple flow metrics
                flow_bytes = float(flow.get('Flow Bytes/s', 0))
                flow_packets = float(flow.get('Flow Packets/s', 0))
                duration = float(flow.get('Flow Duration', 1))
                
                # Composite weight representing connection strength
                weight = (flow_bytes + flow_packets) / max(duration, 1)
                
                if G.has_edge(src_ip, dst_ip):
                    G[src_ip][dst_ip]['weight'] += weight
                    G[src_ip][dst_ip]['flow_count'] += 1
                else:
                    G.add_edge(src_ip, dst_ip, weight=weight, flow_count=1)
                    
            except Exception as e:
                # Skip problematic flows
                continue
        
        # Filter out nodes with too few connections (noise reduction)
        nodes_to_remove = [node for node in G.nodes() 
                          if G.degree(node) < self.min_connections]
        G.remove_nodes_from(nodes_to_remove)
        
        return G

    def extract_graph_tda_features(self, graph_sequences, scale_idx):
        """Extract TDA features from network graph sequences."""
        
        try:
            # Initialize TDA analyzer for this scale
            if scale_idx not in self.tda_analyzers:
                # Adjust TDA parameters based on expected graph size
                max_dim = 1  # Focus on H0 (components) and H1 (cycles)
                thresh = 2.0  # Distance threshold for graph analysis
                
                self.tda_analyzers[scale_idx] = PersistentHomologyAnalyzer(
                    maxdim=max_dim,
                    thresh=thresh,
                    metric='precomputed',  # Use precomputed distance matrix
                    backend='ripser'
                )
            
            ph_analyzer = self.tda_analyzers[scale_idx]
            
            # Process graphs in batches
            batch_size = 20
            all_features = []
            
            for i in range(0, len(graph_sequences), batch_size):
                batch_graphs = graph_sequences[i:i+batch_size]
                batch_features = []
                
                for G in batch_graphs:
                    try:
                        # Convert graph to distance matrix for TDA
                        distance_matrix = self.graph_to_distance_matrix(G)
                        
                        if distance_matrix is not None:
                            # Apply persistent homology
                            ph_analyzer.fit(distance_matrix)
                            features = ph_analyzer.extract_features()
                            
                            # Ensure consistent feature length
                            if len(features) < 12:  # Expected: 6 for H0 + 6 for H1
                                padded_features = np.zeros(12)
                                padded_features[:len(features)] = features
                                features = padded_features
                            
                            # Add graph-specific features
                            graph_features = self.extract_graph_statistics(G)
                            combined_features = np.concatenate([
                                features[:12], graph_features
                            ])
                            
                            batch_features.append(combined_features)
                        else:
                            # Fallback for problematic graphs
                            batch_features.append(np.zeros(18))  # 12 TDA + 6 graph stats
                            
                    except Exception as e:
                        # Fallback for failed TDA computation
                        batch_features.append(np.zeros(18))
                
                if batch_features:
                    all_features.extend(batch_features)
            
            if all_features:
                feature_matrix = np.array(all_features)
                
                # Handle NaN/inf values
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                
                return feature_matrix
            else:
                return None
                
        except Exception as e:
            print(f"      ❌ Error in graph TDA feature extraction: {e}")
            return None

    def graph_to_distance_matrix(self, G):
        """Convert network graph to distance matrix for TDA."""
        
        try:
            if G.number_of_nodes() < 3:
                return None
            
            # Use shortest path distances between nodes
            # This captures the network topology structure
            nodes = list(G.nodes())
            n_nodes = len(nodes)
            
            if n_nodes > 50:  # Limit size for computational efficiency
                # Sample key nodes (highest degree)
                node_degrees = [(node, G.degree(node)) for node in nodes]
                node_degrees.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = [node for node, _ in node_degrees[:50]]
            else:
                selected_nodes = nodes
            
            # Create subgraph with selected nodes
            subG = G.subgraph(selected_nodes)
            
            # Compute shortest path distance matrix
            distance_dict = dict(nx.all_pairs_shortest_path_length(subG))
            
            # Convert to matrix format
            n_selected = len(selected_nodes)
            distance_matrix = np.full((n_selected, n_selected), np.inf)
            
            for i, node_i in enumerate(selected_nodes):
                for j, node_j in enumerate(selected_nodes):
                    if node_j in distance_dict.get(node_i, {}):
                        distance_matrix[i, j] = distance_dict[node_i][node_j]
                    elif i == j:
                        distance_matrix[i, j] = 0
            
            # Replace infinite distances with max finite distance + 1
            max_finite = np.max(distance_matrix[np.isfinite(distance_matrix)])
            distance_matrix[np.isinf(distance_matrix)] = max_finite + 1
            
            return distance_matrix
            
        except Exception as e:
            return None

    def extract_graph_statistics(self, G):
        """Extract basic graph statistics as additional features."""
        
        try:
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            
            if n_nodes == 0:
                return np.zeros(6)
            
            # Basic connectivity metrics
            density = nx.density(G)
            
            # Centrality measures (mean values)
            degree_centrality = np.mean(list(nx.degree_centrality(G).values()))
            
            try:
                betweenness_centrality = np.mean(list(nx.betweenness_centrality(G).values()))
            except:
                betweenness_centrality = 0.0
                
            try:
                clustering = nx.average_clustering(G)
            except:
                clustering = 0.0
            
            # Connectivity metrics
            try:
                n_components = nx.number_connected_components(G)
            except:
                n_components = 1
                
            # Average degree
            avg_degree = np.mean([d for n, d in G.degree()]) if n_nodes > 0 else 0
            
            return np.array([
                density, degree_centrality, betweenness_centrality,
                clustering, n_components / max(n_nodes, 1), avg_degree / max(n_nodes, 1)
            ])
            
        except Exception as e:
            return np.zeros(6)

def load_and_prepare_data():
    """Load the infiltration dataset."""
    
    print("🔍 LOADING INFILTRATION DATASET FOR GRAPH ANALYSIS")
    print("=" * 50)
    
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Get attacks + larger benign sample for graph analysis
    attacks = df[df['Label'] != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"   Total attacks: {len(attacks)}")
    print(f"   Total benign: {len(benign):,}")
    
    # Use larger sample for graph analysis (need more data for meaningful graphs)
    benign_sample = benign.sample(n=min(10000, len(benign)), random_state=42)
    df_balanced = pd.concat([attacks, benign_sample]).sort_index()
    
    print(f"   Using {len(attacks)} attacks + {len(benign_sample):,} benign = {len(df_balanced):,} total")
    
    # Ensure we have the required columns
    required_cols = ['Source IP', 'Destination IP', 'Flow Bytes/s', 'Flow Packets/s', 'Flow Duration']
    missing_cols = [col for col in required_cols if col not in df_balanced.columns]
    
    if missing_cols:
        print(f"   ⚠️ Missing columns: {missing_cols}")
        # Create proxy columns if needed
        for col in missing_cols:
            if col == 'Source IP':
                df_balanced[col] = df_balanced.index % 100  # Proxy IP addresses
            elif col == 'Destination IP':
                df_balanced[col] = (df_balanced.index + 50) % 100
            else:
                df_balanced[col] = np.random.normal(100, 50, len(df_balanced))
    
    print(f"   Final dataset for graph analysis: {df_balanced.shape}")
    print(f"   Attack rate: {(df_balanced['Label'] != 'BENIGN').mean():.3%}")
    
    return df_balanced

def evaluate_graph_based_tda():
    """Main evaluation function for graph-based TDA."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Initialize graph-based analyzer
    analyzer = NetworkGraphTDAAnalyzer()
    
    # Extract graph-based TDA features
    start_time = time.time()
    graph_features, graph_labels = analyzer.extract_network_graphs(df)
    extraction_time = time.time() - start_time
    
    if graph_features is None:
        print("❌ Graph feature extraction failed")
        return None
    
    print(f"\n⏱️ Graph feature extraction completed in {extraction_time:.1f}s")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        graph_features, graph_labels, test_size=0.3, random_state=42, stratify=graph_labels
    )
    
    print(f"\n📊 EVALUATION SETUP:")
    print(f"   Training: {X_train.shape}, attacks: {y_train.sum()}")
    print(f"   Testing: {X_test.shape}, attacks: {y_test.sum()}")
    
    # Train classifier on graph TDA features
    print(f"\n🚀 TRAINING GRAPH-BASED TDA CLASSIFIER")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest optimized for graph features
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluation
    print(f"\n📈 GRAPH-BASED TDA RESULTS:")
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
    
    # Handle confusion matrix display
    if cm.shape == (2, 2):
        print(f"\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    else:
        print(f"\n   Confusion Matrix: {cm}")
        print(f"   (Limited class representation in test data)")
    
    # Compare with previous results
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print("=" * 50)
    
    multi_scale_f1 = 0.654  # Previous multi-scale TDA result
    improvement = f1_score - multi_scale_f1
    improvement_pct = (improvement / multi_scale_f1) * 100 if multi_scale_f1 > 0 else 0
    
    print(f"   Multi-Scale TDA (flow-based): F1 = {multi_scale_f1:.3f}")
    print(f"   Graph-Based TDA: F1 = {f1_score:.3f}")
    print(f"   Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    # Target assessment
    target_f1 = 0.75  # Conservative target from enhancement strategy
    if f1_score > target_f1:
        print(f"   ✅ SUCCESS: Exceeded Phase 2A target (F1 > 75%)")
    elif f1_score > multi_scale_f1:
        print(f"   ✅ PROGRESS: Improvement over multi-scale baseline")
    else:
        print(f"   ⚠️ MIXED: Performance assessment needed")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'confusion_matrix': cm,
        'extraction_time': extraction_time
    }

def main():
    """Main execution function."""
    
    print("🕸️ GRAPH-BASED NETWORK TOPOLOGY TDA IMPLEMENTATION")
    print("=" * 70)
    print("Phase 2A of Advanced TDA Enhancement Strategy")
    print("Target: F1-Score >75% (current multi-scale: 65.4%)")
    print("=" * 70)
    
    # Run evaluation
    results = evaluate_graph_based_tda()
    
    if results:
        print(f"\n🎯 PHASE 2A EVALUATION COMPLETE")
        print("=" * 70)
        
        if results['f1_score'] > 0.75:
            print(f"✅ SUCCESS: Graph-based TDA achieved target performance!")
            print(f"   Recommended: Proceed to Phase 2B (Temporal Evolution)")
        elif results['improvement'] > 0:
            print(f"⚠️ PARTIAL SUCCESS: Improvement achieved but target not met")
            print(f"   Recommended: Optimize parameters and consider hybrid approach")
        else:
            print(f"❌ CHALLENGE: Graph approach needs refinement")
            print(f"   Recommended: Debug graph construction or try alternative features")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Document results in EXPERIMENT_LOG.md")
        print(f"   2. Analyze feature importance to understand graph topology impact")
        print(f"   3. Begin Phase 2B: Temporal Persistence Evolution if successful")

if __name__ == "__main__":
    main()