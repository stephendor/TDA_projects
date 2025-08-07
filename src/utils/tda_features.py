import numpy as np
from sklearn.preprocessing import StandardScaler
import ripser
from sklearn.neighbors import kneighbors_graph

def compute_single_persistence_diagram(X_sample, max_dimension=2):
    """Compute persistence diagram for a single sample"""
    if len(X_sample.shape) == 1:
        X_sample = X_sample.reshape(1, -1)
    
    if len(X_sample) < 3:
        # Not enough points for meaningful topology
        return [np.array([[0.0, 0.0]]) for _ in range(max_dimension + 1)]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # Create k-NN graph to improve topology detection
    k = min(5, len(X_sample) - 1)
    knn_graph = kneighbors_graph(X_scaled, n_neighbors=k, mode='distance', include_self=False)
    
    # Use lower threshold to enable H1, H2 features
    rips = ripser.Rips(maxdim=max_dimension, thresh=1.5)
    
    try:
        diagrams = rips.fit_transform(X_scaled)
        return diagrams
    except:
        # Fallback: return empty diagrams
        return [np.array([[0.0, 0.0]]) for _ in range(max_dimension + 1)]

def extract_persistence_features(diagrams, max_dimension=2):
    """Extract comprehensive features from persistence diagrams"""
    features = []
    
    for dim in range(max_dimension + 1):
        if dim < len(diagrams) and len(diagrams[dim]) > 0:
            diagram = diagrams[dim]
            # Filter out infinite points
            finite_mask = diagram[:, 1] != np.inf
            finite_diagram = diagram[finite_mask]
            
            if len(finite_diagram) > 0:
                persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
                
                features.extend([
                    len(finite_diagram),          # Number of finite features
                    np.max(persistences),         # Max persistence
                    np.mean(persistences),        # Mean persistence  
                    np.std(persistences),         # Std persistence
                    np.sum(persistences),         # Total persistence
                    np.median(persistences),      # Median persistence
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Count infinite features
            inf_count = np.sum(diagram[:, 1] == np.inf)
            features.append(inf_count)
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])  # 7 features per dimension
    
    return np.array(features)
