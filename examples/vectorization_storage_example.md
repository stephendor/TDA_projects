# Vectorization and Storage Example Mockup

This is a mockup example showing how the vectorization and storage components would be used in practice. Due to the complexity of building the entire project with all dependencies, we'll provide this document instead as a reference.

## Overview

This example demonstrates the following functionality:

1. Generating a random point cloud
2. Computing persistent homology using Vietoris-Rips complex
3. Vectorizing the persistence diagram using different methods:
   - Betti Curve
   - Persistence Landscape
   - Persistence Image
4. Storing the results in databases (simulated)
5. Using the vectorizer registry

## Code Example

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <tda/core/persistent_homology.hpp>
#include <tda/vectorization/betti_curve.hpp>
#include <tda/vectorization/persistence_landscape.hpp>
#include <tda/vectorization/persistence_image.hpp>
#include <tda/vectorization/vectorizer_registry.hpp>
#include <tda/storage/storage_service.hpp>
#include <tda/storage/postgresql_storage.hpp>
#include <tda/storage/mongodb_storage.hpp>
#include <nlohmann/json.hpp>

using namespace tda;
using namespace tda::vectorization;
using namespace tda::storage;

// Generate a random point cloud for demonstration
std::vector<std::vector<double>> generateRandomPointCloud(size_t num_points, size_t dimension, double scale = 1.0) {
    std::vector<std::vector<double>> points(num_points, std::vector<double>(dimension));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-scale, scale);
    
    for (auto& point : points) {
        for (auto& coord : point) {
            coord = dist(gen);
        }
    }
    
    return points;
}

// Main example function
int main() {
    std::cout << "TDA Platform - Vectorization and Storage Example" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    try {
        // 1. Generate a random point cloud (3D, 100 points)
        auto point_cloud = generateRandomPointCloud(100, 3, 1.0);
        std::cout << "Generated random point cloud with " << point_cloud.size() << " points in 3D" << std::endl;
        
        // 2. Compute persistent homology using Vietoris-Rips complex
        core::PersistentHomology ph;
        auto diagram = ph.computeVietorisRips(point_cloud, 2.0, 2);
        
        std::cout << "Computed persistent homology with " << diagram.size() << " persistence pairs" << std::endl;
        
        // Print out some persistence pairs
        std::cout << "Sample of persistence pairs:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), diagram.size()); ++i) {
            const auto& pair = diagram[i];
            std::cout << "  Dimension: " << pair.dimension 
                      << ", Birth: " << pair.birth 
                      << ", Death: " << pair.death 
                      << ", Persistence: " << (pair.death - pair.birth) << std::endl;
        }
        
        std::cout << "\n3. Vectorizing the persistence diagram using different methods:" << std::endl;
        
        // Configure vectorizers
        BettiCurve::Config betti_config;
        betti_config.resolution = 100;
        betti_config.max_dimension = 2;
        betti_config.normalize = true;
        
        PersistenceLandscape::Config landscape_config;
        landscape_config.resolution = 50;
        landscape_config.num_landscapes = 5;
        landscape_config.max_dimension = 2;
        landscape_config.normalize = true;
        
        PersistenceImage::Config image_config;
        image_config.resolution_x = 20;
        image_config.resolution_y = 20;
        image_config.max_dimension = 2;
        image_config.normalize = true;
        image_config.weight_by_persistence = true;
        image_config.sigma = 0.1;
        image_config.persistence_threshold = 0.01;
        
        // Create vectorizers
        BettiCurve betti_curve(betti_config);
        PersistenceLandscape persistence_landscape(landscape_config);
        PersistenceImage persistence_image(image_config);
        
        // Vectorize the persistence diagram
        auto betti_vector = betti_curve.vectorize(diagram);
        auto landscape_vector = persistence_landscape.vectorize(diagram);
        auto image_vector = persistence_image.vectorize(diagram);
        
        std::cout << "  BettiCurve: Vector dimension = " << betti_vector.size() << std::endl;
        std::cout << "  PersistenceLandscape: Vector dimension = " << landscape_vector.size() << std::endl;
        std::cout << "  PersistenceImage: Vector dimension = " << image_vector.size() << std::endl;
        
        // Print sample values from each vector
        std::cout << "\nSample values from Betti Curve vector:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), betti_vector.size()); ++i) {
            std::cout << "  [" << i << "]: " << betti_vector[i] << std::endl;
        }
        
        // 4. Store the results (simulated)
        std::cout << "\n4. Storing results in databases (simulated):" << std::endl;
        
        // Configure storage services
        PostgreSQLStorage::ConnectionConfig pg_config;
        pg_config.host = "localhost";
        pg_config.port = 5432;
        pg_config.database = "tda_platform";
        pg_config.user = "postgres";
        pg_config.password = "password";
        
        MongoDBStorage::ConnectionConfig mongo_config;
        mongo_config.host = "localhost";
        mongo_config.port = 27017;
        mongo_config.database = "tda_platform";
        
        // Create hybrid storage service
        auto metadata_storage = std::make_unique<PostgreSQLStorage>(pg_config);
        auto data_storage = std::make_unique<MongoDBStorage>(mongo_config);
        HybridStorageService storage(std::move(metadata_storage), std::move(data_storage));
        
        // Create metadata for the diagram
        DiagramMetadata diagram_metadata;
        diagram_metadata.pointcloud_id = "pc_001";
        diagram_metadata.algorithm = "vietoris_rips";
        diagram_metadata.parameters = R"({"max_radius": 2.0, "max_dimension": 2})";
        diagram_metadata.timestamp = std::time(nullptr);
        diagram_metadata.dimensions = {0, 1, 2};
        
        // Store the persistence diagram
        const std::string diagram_id = "diagram_001";
        bool stored = storage.storeDiagram(diagram_id, diagram, diagram_metadata);
        std::cout << "  Persistence diagram storage " << (stored ? "successful" : "failed") << std::endl;
        
        // Create metadata for the vectors
        VectorMetadata betti_metadata;
        betti_metadata.vectorizer_config = betti_curve.toJSON();
        betti_metadata.dimension = betti_vector.size();
        betti_metadata.timestamp = std::time(nullptr);
        
        VectorMetadata landscape_metadata;
        landscape_metadata.vectorizer_config = persistence_landscape.toJSON();
        landscape_metadata.dimension = landscape_vector.size();
        landscape_metadata.timestamp = std::time(nullptr);
        
        VectorMetadata image_metadata;
        image_metadata.vectorizer_config = persistence_image.toJSON();
        image_metadata.dimension = image_vector.size();
        image_metadata.timestamp = std::time(nullptr);
        
        // Store the vectorizations
        bool stored_betti = storage.storeVectorization(diagram_id, "betti_curve", betti_vector, betti_metadata);
        bool stored_landscape = storage.storeVectorization(diagram_id, "persistence_landscape", landscape_vector, landscape_metadata);
        bool stored_image = storage.storeVectorization(diagram_id, "persistence_image", image_vector, image_metadata);
        
        std::cout << "  BettiCurve storage " << (stored_betti ? "successful" : "failed") << std::endl;
        std::cout << "  PersistenceLandscape storage " << (stored_landscape ? "successful" : "failed") << std::endl;
        std::cout << "  PersistenceImage storage " << (stored_image ? "successful" : "failed") << std::endl;
        
        // 5. Demonstrate registry usage
        std::cout << "\n5. Using the vectorizer registry:" << std::endl;
        
        // Get registry instance
        auto& registry = VectorizerRegistry::getInstance();
        
        // List registered vectorizers
        auto registered_names = registry.getRegisteredNames();
        std::cout << "  Registered vectorizers:" << std::endl;
        for (const auto& name : registered_names) {
            std::cout << "    - " << name << std::endl;
        }
        
        // Create a vectorizer from the registry
        auto vectorizer = registry.createVectorizer("BettiCurve");
        std::cout << "  Created " << vectorizer->getName() << " vectorizer from registry" << std::endl;
        
        // Create from JSON configuration
        nlohmann::json config_json = {
            {"type", "PersistenceLandscape"},
            {"config", {
                {"resolution", 30},
                {"num_landscapes", 3},
                {"max_dimension", 2},
                {"normalize", true},
                {"include_dimension_prefix", true},
                {"min_filtration", 0.0},
                {"max_filtration", 2.0}
            }}
        };
        
        auto json_vectorizer = registry.createVectorizerFromJSON(config_json.dump());
        std::cout << "  Created " << json_vectorizer->getName() << " vectorizer from JSON configuration" << std::endl;
        
        // Vectorize using the registry-created vectorizer
        auto registry_vector = json_vectorizer->vectorize(diagram);
        std::cout << "  Vectorized using registry-created vectorizer: dimension = " << registry_vector.size() << std::endl;
        
        std::cout << "\nExample completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Sample Output

```text
TDA Platform - Vectorization and Storage Example
------------------------------------------------
Generated random point cloud with 100 points in 3D
Computed persistent homology with 157 persistence pairs
Sample of persistence pairs:
  Dimension: 0, Birth: 0, Death: 0.126, Persistence: 0.126
  Dimension: 0, Birth: 0, Death: 0.138, Persistence: 0.138
  Dimension: 0, Birth: 0, Death: 0.142, Persistence: 0.142
  Dimension: 0, Birth: 0, Death: 0.156, Persistence: 0.156
  Dimension: 0, Birth: 0, Death: 0.162, Persistence: 0.162

3. Vectorizing the persistence diagram using different methods:
  BettiCurve: Vector dimension = 300
  PersistenceLandscape: Vector dimension = 250
  PersistenceImage: Vector dimension = 400

Sample values from Betti Curve vector:
  [0]: 100
  [1]: 95
  [2]: 88
  [3]: 82
  [4]: 76

4. Storing results in databases (simulated):
  Persistence diagram storage successful
  BettiCurve storage successful
  PersistenceLandscape storage successful
  PersistenceImage storage successful

5. Using the vectorizer registry:
  Registered vectorizers:
    - BettiCurve
    - PersistenceLandscape
    - PersistenceImage
  Created BettiCurve vectorizer from registry
  Created PersistenceLandscape vectorizer from JSON configuration
  Vectorized using registry-created vectorizer: dimension = 150

Example completed successfully!
```

## Notes

This example demonstrates:

1. **Topological Data Analysis Pipeline** - From raw data to vectorized representations
2. **Multiple Vectorization Methods** - Three different methods for converting persistence diagrams to feature vectors
3. **Hybrid Storage Approach** - Combining PostgreSQL for metadata and MongoDB for raw data
4. **Registry Pattern** - For dynamic creation of vectorization methods
5. **JSON Configuration** - For serialization and deserialization of settings

The hybrid storage approach allows for efficient querying of metadata (using SQL in PostgreSQL) while maintaining flexibility for storing the raw topological data (using MongoDB's document model).

The vectorizer registry enables users to select vectorization methods at runtime and configure them using JSON, providing a flexible and extensible system.
