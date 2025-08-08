import pandas as pd
import numpy as np
from pathlib import Path

from src.datasets.data_ingestion import DataIngestion
from src.algorithms.tda_features import TDAFeatureGenerator
from src.algorithms.tda_vectorization import TDAVectorizer

class TDAPipeline:
    def __init__(self, chunk_size=100000):
        self.data_ingestion = DataIngestion()
        self.tda_feature_generator = TDAFeatureGenerator()
        self.tda_vectorizer = TDAVectorizer()
        self.chunk_size = chunk_size

    def _preprocess_chunk(self, df_chunk):
        """
        Preprocesses a single data chunk for TDA. Handles missing values and selects numeric features.
        """
        # Drop rows with any NaN values for simplicity in TDA processing
        df_chunk = df_chunk.dropna()

        # Identify label column (case-insensitive)
        label_col = None
        for col in df_chunk.columns:
            if 'label' in col.lower():
                label_col = col
                break

        if label_col:
            labels = df_chunk[label_col].apply(lambda x: 1 if x != 'BENIGN' else 0).values
            features_df = df_chunk.drop(columns=[label_col])
        else:
            labels = np.zeros(len(df_chunk)) # Default to benign if no label column
            features_df = df_chunk

        # Select only numeric features for TDA
        numeric_features_df = features_df.select_dtypes(include=[np.number])

        if numeric_features_df.empty:
            print("Warning: No numeric features found in chunk after preprocessing.")
            return None, None

        return numeric_features_df.values, labels

    def process_dataset(self, dataset_path, feature_vector_method="persistence_image"):
        """
        Processes a dataset in chunks, generating and vectorizing TDA features.

        Args:
            dataset_path (Path): Absolute path to the dataset (e.g., Parquet file).
            feature_vector_method (str): Method to vectorize persistence diagrams.

        Yields:
            tuple: (vectorized_tda_features, labels) for each chunk.
        """
        print(f"\nStarting TDA pipeline for dataset: {dataset_path}")
        all_vectorized_features = []
        all_labels = []

        for i, chunk_df in enumerate(self.data_ingestion.load_parquet_chunks(dataset_path, self.chunk_size)):
            if chunk_df is None or chunk_df.empty:
                print(f"Skipping empty or invalid chunk {i+1}.")
                continue

            print(f"Processing chunk {i+1} with shape: {chunk_df.shape}")
            data_points, labels = self._preprocess_chunk(chunk_df)

            if data_points is None or data_points.size == 0:
                print(f"No valid data points in chunk {i+1} after preprocessing. Skipping.")
                continue

            # Generate Persistence Diagrams
            # For simplicity, we'll generate diagrams for each row as a point cloud of 1 point
            # In a real scenario, you'd define how to form point clouds from your data chunks
            # e.g., sliding windows, network graphs, etc.
            # Here, we'll treat each row as a point in a high-dimensional space for demonstration.
            # Or, more realistically, we'd expect `data_points` to be a collection of point clouds.
            # For now, let's assume `data_points` is already a collection of point clouds or can be treated as such.
            # Let's simplify: treat each row as a point, and generate a diagram for a small sample of rows
            # or for a window of rows if we were doing time series.

            # For demonstration, let's take a small sample from the chunk to form a single point cloud
            # and generate one persistence diagram per chunk. This is a simplification.
            # A more robust approach would involve creating multiple point clouds per chunk.
            sample_size = min(500, data_points.shape[0]) # Take up to 500 points from the chunk
            if sample_size == 0:
                print(f"Chunk {i+1} has no data points after sampling. Skipping.")
                continue
            
            # Randomly sample points if the chunk is too large for direct TDA
            if data_points.shape[0] > sample_size:
                indices = np.random.choice(data_points.shape[0], sample_size, replace=False)
                point_cloud = data_points[indices]
            else:
                point_cloud = data_points

            try:
                persistence_diagrams = self.tda_feature_generator.generate_persistence_diagrams(point_cloud, homology_dimensions=[0, 1])
            except Exception as e:
                print(f"Error generating persistence diagrams for chunk {i+1}: {e}. Skipping chunk.")
                continue

            vectorized_features_chunk = []
            for dim, diagram in persistence_diagrams.items():
                if diagram.size > 0:
                    # Vectorize each dimension's diagram
                    vec = self.tda_vectorizer.vectorize_persistence_diagram(diagram, method=feature_vector_method)
                    vectorized_features_chunk.append(vec)
                else:
                    # Append zeros if diagram is empty for a dimension
                    # Need to know the expected size of the vectorized output for empty diagrams
                    # For persistence_image, it's resolution[0] * resolution[1]
                    # Let's assume a default resolution of [20,20] for now
                    default_vec_size = 20 * 20 # Default for persistence_image
                    if feature_vector_method == "persistence_landscape":
                        default_vec_size = 100 # From placeholder
                    elif feature_vector_method == "betti_curve":
                        default_vec_size = 50 # From placeholder
                    vectorized_features_chunk.append(np.zeros(default_vec_size))

            if vectorized_features_chunk:
                # Concatenate features from different homology dimensions
                final_vectorized_features = np.concatenate(vectorized_features_chunk)
                all_vectorized_features.append(final_vectorized_features)
                # For simplicity, we'll associate the labels of the *entire* chunk with this single TDA feature vector.
                # In a real application, you'd need a more sophisticated mapping from point cloud to labels.
                # Here, we'll just take the majority label or a representative label from the chunk.
                if labels.size > 0:
                    all_labels.append(np.bincount(labels).argmax()) # Majority vote for chunk label
                else:
                    all_labels.append(0) # Default to benign if no labels

        if all_vectorized_features:
            yield np.array(all_vectorized_features), np.array(all_labels)
        else:
            yield np.array([]), np.array([])

if __name__ == "__main__":
    pipeline = TDAPipeline(chunk_size=10000) # Process in smaller chunks for demonstration

    # --- UNSW-NB15 Example ---
    print("\n=== Processing UNSW-NB15 Data ===")
    unsw_paths = pipeline.data_ingestion.get_unsw_nb15_data()
    if unsw_paths and unsw_paths["training"].exists():
        for vectorized_features, labels in pipeline.process_dataset(unsw_paths["training"]):
            if vectorized_features.size > 0:
                print(f"Successfully processed UNSW-NB15 training data.")
                print(f"  Vectorized Features Shape: {vectorized_features.shape}")
                print(f"  Labels Shape: {labels.shape}")
                print(f"  Sample Label Distribution: {np.unique(labels, return_counts=True)}")
            else:
                print("No vectorized features generated for UNSW-NB15 training data.")
    else:
        print("UNSW-NB15 training data not available. Please ensure it's downloaded and converted to Parquet.")

    # --- CIC-IDS-2017 Example (Placeholder) ---
    print("\n=== Processing CIC-IDS-2017 Data (Placeholder) ===")
    cic_paths = pipeline.data_ingestion.get_cic_ids_2017_data()
    if cic_paths:
        # Process one of the CIC-IDS-2017 files, e.g., Monday's data
        monday_path = cic_paths.get('Monday-WorkingHours.pcap_ISCX')
        if monday_path and monday_path.exists():
            for vectorized_features, labels in pipeline.process_dataset(monday_path):
                if vectorized_features.size > 0:
                    print(f"Successfully processed CIC-IDS-2017 Monday data.")
                    print(f"  Vectorized Features Shape: {vectorized_features.shape}")
                    print(f"  Labels Shape: {labels.shape}")
                    print(f"  Sample Label Distribution: {np.unique(labels, return_counts=True)}")
                else:
                    print("No vectorized features generated for CIC-IDS-2017 Monday data.")
        else:
            print("CIC-IDS-2017 Monday data not available. Please ensure it's downloaded and converted to Parquet.")
    else:
        print("CIC-IDS-2017 data not available. Please ensure it's downloaded and converted to Parquet.")

    # --- APT/Netflow Example (Placeholder) ---
    print("\n=== Processing APT/Netflow Data (Placeholder) ===")
    apt_netflow_path = pipeline.data_ingestion.get_apt_netflow_data()
    if apt_netflow_path and apt_netflow_path.exists():
        for vectorized_features, labels in pipeline.process_dataset(apt_netflow_path):
            if vectorized_features.size > 0:
                print(f"Successfully processed APT/Netflow data.")
                print(f"  Vectorized Features Shape: {vectorized_features.shape}")
                print(f"  Labels Shape: {labels.shape}")
                print(f"  Sample Label Distribution: {np.unique(labels, return_counts=True)}")
            else:
                print("No vectorized features generated for APT/Netflow data.")
    else:
        print("APT/Netflow data not available. Please ensure it's downloaded and converted to Parquet.")
