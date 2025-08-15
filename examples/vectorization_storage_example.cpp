#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <tda/core/persistent_homology.hpp>
#include <tda/algorithms/vietoris_rips.hpp>
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
    // Build Vietoris-Rips and compute persistence
    algorithms::VietorisRips vr;
    auto r1 = vr.initialize(point_cloud, 2.0, 2);
    if (r1.has_error()) throw std::runtime_error(r1.error());
    auto r2 = vr.computeComplex();
    if (r2.has_error()) throw std::runtime_error(r2.error());
    auto r3 = vr.computePersistence();
    if (r3.has_error()) throw std::runtime_error(r3.error());
    auto pairs_res = vr.getPersistencePairs();
    if (pairs_res.has_error()) throw std::runtime_error(pairs_res.error());
    auto diagram = pairs_res.value();
        
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
    // Weighting is built-in via default weighting function; single isotropic sigma is used
    image_config.sigma = 0.1;
        
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
        
    auto json_vectorizer = registry.createVectorizer("PersistenceLandscape");
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
