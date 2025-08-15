#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <nlohmann/json.hpp>

namespace tda {
namespace storage {

/**
 * @brief Database schema definitions for topological feature storage
 * 
 * This file defines the schema for storing persistence diagrams and their
 * vectorized representations in a database system. It uses a hybrid approach
 * with PostgreSQL for metadata and MongoDB for raw data.
 */

/**
 * @brief Origin information for a persistence diagram
 * 
 * Stores information about where a persistence diagram came from,
 * such as the input dataset, filtration method, and parameters.
 */
struct DiagramOrigin {
    // Unique identifier for the dataset
    std::string dataset_id;
    
    // Method used to generate the diagram (e.g., "alpha_complex", "vietoris_rips")
    std::string method;
    
    // Method-specific parameters as JSON
    nlohmann::json parameters;
    
    // Human-readable description
    std::string description;
    
    // Creation timestamp
    std::chrono::system_clock::time_point created_at;
};

/**
 * @brief Metadata for a persistence diagram
 * 
 * Stores metadata about a persistence diagram, such as its ID,
 * versioning information, and statistics.
 */
struct DiagramMetadata {
    // Unique identifier for this diagram
    std::string id;
    
    // Origin information
    DiagramOrigin origin;
    
    // Version information
    std::string version;
    int version_number;
    
    // Statistics about the diagram
    int max_dimension;
    size_t total_pairs;
    std::vector<size_t> pairs_per_dimension;
    double min_persistence;
    double max_persistence;
    
    // Creation and last update timestamps
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    // Optional tags for categorization
    std::vector<std::string> tags;
};

/**
 * @brief Vector feature metadata
 * 
 * Stores metadata about a vector representation of a persistence diagram,
 * such as the vectorization method and parameters.
 */
struct VectorMetadata {
    // Unique identifier for this vector
    std::string id;
    
    // Reference to the persistence diagram
    std::string diagram_id;
    
    // Vectorization method used (e.g., "betti_curve", "persistence_landscape")
    std::string method;
    
    // Method-specific parameters as JSON
    nlohmann::json parameters;
    
    // Dimension of the vector
    size_t dimension;
    
    // Creation timestamp
    std::chrono::system_clock::time_point created_at;
    
    // Optional tags for categorization
    std::vector<std::string> tags;
};

/**
 * @brief PostgreSQL schema definition
 * 
 * Defines the tables, indexes, and constraints for the PostgreSQL database
 * that stores metadata for persistence diagrams and their vectorizations.
 */
struct PostgreSQLSchema {
    static constexpr const char* DIAGRAM_METADATA_TABLE = "diagram_metadata";
    static constexpr const char* VECTOR_METADATA_TABLE = "vector_metadata";
    static constexpr const char* TAGS_TABLE = "tags";
    static constexpr const char* DIAGRAM_TAGS_TABLE = "diagram_tags";
    static constexpr const char* VECTOR_TAGS_TABLE = "vector_tags";
    
    /**
     * @brief Get the SQL statements to create the schema
     * 
     * @return std::vector<std::string> SQL statements
     */
    static std::vector<std::string> getCreateStatements();
    
    /**
     * @brief Get the SQL statements to drop the schema
     * 
     * @param cascade Whether to use CASCADE when dropping
     * @return std::vector<std::string> SQL statements
     */
    static std::vector<std::string> getDropStatements(bool cascade = false);
    
    /**
     * @brief Get the SQL statements to create the indexes
     * 
     * @return std::vector<std::string> SQL statements
     */
    static std::vector<std::string> getCreateIndexStatements();
};

/**
 * @brief MongoDB schema definition
 * 
 * Defines the collections and indexes for the MongoDB database
 * that stores raw persistence diagrams and vector features.
 */
struct MongoDBSchema {
    static constexpr const char* DIAGRAM_COLLECTION = "persistence_diagrams";
    static constexpr const char* VECTOR_COLLECTION = "feature_vectors";
    
    /**
     * @brief Get the JSON schema for the diagram collection
     * 
     * @return nlohmann::json JSON schema
     */
    static nlohmann::json getDiagramSchema();
    
    /**
     * @brief Get the JSON schema for the vector collection
     * 
     * @return nlohmann::json JSON schema
     */
    static nlohmann::json getVectorSchema();
    
    /**
     * @brief Get the index definitions for the diagram collection
     * 
     * @return std::vector<nlohmann::json> Index definitions
     */
    static std::vector<nlohmann::json> getDiagramIndexes();
    
    /**
     * @brief Get the index definitions for the vector collection
     * 
     * @return std::vector<nlohmann::json> Index definitions
     */
    static std::vector<nlohmann::json> getVectorIndexes();
};

} // namespace storage
} // namespace tda
