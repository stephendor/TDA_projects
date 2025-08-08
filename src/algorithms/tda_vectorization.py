import numpy as np
from persim import PersistenceImager

class TDAVectorizer:
    def __init__(self):
        pass

    def vectorize_persistence_diagram(self, diagram, method="persistence_image", **kwargs):
        """
        Vectorizes a single persistence diagram using a specified method.

        Args:
            diagram (np.ndarray): A 2D numpy array representing the persistence diagram
                                  (birth, death) pairs.
            method (str): The vectorization method to use. Options: "persistence_image",
                          "persistence_landscape" (placeholder), "betti_curve" (placeholder).
            **kwargs: Additional arguments for the vectorization method.

        Returns:
            np.ndarray: A 1D numpy array representing the vectorized persistence diagram.
        """
        if method == "persistence_image":
            return self._persistence_image_vectorization(diagram, **kwargs)
        elif method == "persistence_landscape":
            return self._persistence_landscape_placeholder(diagram, **kwargs)
        elif method == "betti_curve":
            return self._betti_curve_placeholder(diagram, **kwargs)
        else:
            raise ValueError(f"Unknown vectorization method: {method}")

    def _persistence_image_vectorization(self, diagram, birth_range=None, pers_range=None, pixel_size=0.2):
        """
        Vectorizes a persistence diagram into a persistence image.

        Args:
            diagram (np.ndarray): Persistence diagram (birth, death) pairs.
            birth_range (tuple): (min_birth, max_birth) for the image. If None, inferred from data.
            pers_range (tuple): (min_persistence, max_persistence) for the image. If None, inferred from data.
            pixel_size (float): Determines the dimensions of each square pixel in the resulting image.

        Returns:
            np.ndarray: A 1D numpy array representing the flattened persistence image.
        """
        # Estimate default output size for empty diagrams based on typical ranges and pixel_size
        # This is a rough estimate, actual size depends on inferred ranges if birth_range/pers_range are None
        estimated_x_range = (birth_range[1] - birth_range[0]) if birth_range else 1.0 # Default range 0-1
        estimated_y_range = (pers_range[1] - pers_range[0]) if pers_range else 1.0 # Default range 0-1
        estimated_res_x = int(estimated_x_range / pixel_size) if pixel_size > 0 else 20
        estimated_res_y = int(estimated_y_range / pixel_size) if pixel_size > 0 else 20
        default_output_size = estimated_res_x * estimated_res_y

        if diagram.size == 0:
            return np.zeros(default_output_size)

        pim = PersistenceImager(birth_range=birth_range, pers_range=pers_range, pixel_size=pixel_size)
        # Convert (birth, death) to (birth, persistence) for PersistenceImager
        birth_persistence_diagram = np.array([[p[0], p[1] - p[0]] for p in diagram])
        
        # Handle cases where persistence is 0 (points on the diagonal)
        # PersistenceImager expects persistence > 0. Filter these out or handle as needed.
        # For now, we'll filter out points with zero persistence as they don't contribute to the image.
        birth_persistence_diagram = birth_persistence_diagram[birth_persistence_diagram[:, 1] > 0]

        if birth_persistence_diagram.size == 0:
            return np.zeros(default_output_size)

        image = pim.transform([birth_persistence_diagram])[0]
        return image.flatten()

    def _persistence_landscape_placeholder(self, diagram, **kwargs):
        """
        Placeholder for Persistence Landscape vectorization.
        Requires a dedicated library or custom implementation (e.g., from giotto-tda).
        """
        print("Persistence Landscape vectorization is a placeholder and requires further implementation.")
        # For demonstration, return a dummy array based on diagram size
        return np.zeros(100) # Example fixed size

    def _betti_curve_placeholder(self, diagram, **kwargs):
        """
        Placeholder for Betti Curve vectorization.
        """
        print("Betti Curve vectorization is a placeholder and requires further implementation.")
        # For demonstration, return a dummy array based on diagram size
        return np.zeros(50) # Example fixed size

if __name__ == "__main__":
    vectorizer = TDAVectorizer()

    # Example Persistence Diagram (birth, death) pairs
    diagram_0 = np.array([
        [0.0, 1.0],
        [0.1, 1.5],
        [0.2, 0.8],
        [0.5, 2.0]
    ])

    diagram_1 = np.array([
        [0.3, 0.7],
        [0.6, 1.2]
    ])

    print("\n--- Vectorizing Diagram 0 with Persistence Image ---")
    vec_0 = vectorizer.vectorize_persistence_diagram(diagram_0, method="persistence_image")
    print(f"Vectorized shape: {vec_0.shape}")
    # print(vec_0) # Uncomment to see the actual vector

    print("\n--- Vectorizing Diagram 1 with Persistence Image (custom resolution) ---")
    vec_1 = vectorizer.vectorize_persistence_diagram(diagram_1, method="persistence_image", resolution=[10, 10])
    print(f"Vectorized shape: {vec_1.shape}")

    print("\n--- Testing Persistence Landscape Placeholder ---")
    vec_pl = vectorizer.vectorize_persistence_diagram(diagram_0, method="persistence_landscape")
    print(f"Vectorized shape: {vec_pl.shape}")

    print("\n--- Testing Betti Curve Placeholder ---")
    vec_bc = vectorizer.vectorize_persistence_diagram(diagram_0, method="betti_curve")
    print(f"Vectorized shape: {vec_bc.shape}")

    print("\n--- Testing Empty Diagram ---")
    empty_diagram = np.array([])
    vec_empty = vectorizer.vectorize_persistence_diagram(empty_diagram, method="persistence_image")
    print(f"Vectorized shape for empty diagram: {vec_empty.shape}")
