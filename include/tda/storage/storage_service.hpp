#pragma once

#include <tda/core/types.hpp>
#include <tda/storage/database_schema.hpp>
#include <tda/vectorization/vectorizer.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <future>

namespace tda {
namespace storage {

/**
 * @brief Abstract storage service interface
 * 
 * This interface defines the operations for storing and retrieving
 * persistence diagrams and their vectorized representations.
 */
class StorageService {
public:
    using PersistenceDiagram = std::vector<tda::core::PersistencePair>;
    using FeatureVector = std::vector<double>;
    
    virtual ~StorageService() = default;
    
    /**
     * @brief Store a persistence diagram
     * 
     * @param diagram The persistence diagram to store
     * @param origin The origin information for the diagram
     * @param tags Optional tags for categorization
     * @return std::string The ID of the stored diagram
     */
    virtual std::string storeDiagram(
        const PersistenceDiagram& diagram,
        const DiagramOrigin& origin,
        const std::vector<std::string>& tags = {}) = 0;
    
    /**
     * @brief Store a vectorized representation of a diagram
     * 
     * @param diagram_id The ID of the related persistence diagram
     * @param vector The feature vector to store
     * @param method The vectorization method used
     * @param parameters The parameters used for vectorization
     * @param tags Optional tags for categorization
     * @return std::string The ID of the stored vector
     */
    virtual std::string storeVector(
        const std::string& diagram_id,
        const FeatureVector& vector,
        const std::string& method,
        const nlohmann::json& parameters,
        const std::vector<std::string>& tags = {}) = 0;
    
    /**
     * @brief Store a vectorized representation of a diagram
     * 
     * @param diagram_id The ID of the related persistence diagram
     * @param vector The feature vector to store
     * @param vectorizer The vectorizer used to create the vector
     * @param tags Optional tags for categorization
     * @return std::string The ID of the stored vector
     */
    virtual std::string storeVector(
        const std::string& diagram_id,
        const FeatureVector& vector,
        const tda::vectorization::Vectorizer& vectorizer,
        const std::vector<std::string>& tags = {}) = 0;
    
    /**
     * @brief Retrieve a persistence diagram by ID
     * 
     * @param id The ID of the diagram to retrieve
     * @return std::optional<PersistenceDiagram> The diagram, or nullopt if not found
     */
    virtual std::optional<PersistenceDiagram> getDiagram(const std::string& id) = 0;
    
    /**
     * @brief Retrieve a feature vector by ID
     * 
     * @param id The ID of the vector to retrieve
     * @return std::optional<FeatureVector> The vector, or nullopt if not found
     */
    virtual std::optional<FeatureVector> getVector(const std::string& id) = 0;
    
    /**
     * @brief Retrieve metadata for a diagram
     * 
     * @param id The ID of the diagram
     * @return std::optional<DiagramMetadata> The metadata, or nullopt if not found
     */
    virtual std::optional<DiagramMetadata> getDiagramMetadata(const std::string& id) = 0;
    
    /**
     * @brief Retrieve metadata for a vector
     * 
     * @param id The ID of the vector
     * @return std::optional<VectorMetadata> The metadata, or nullopt if not found
     */
    virtual std::optional<VectorMetadata> getVectorMetadata(const std::string& id) = 0;
    
    /**
     * @brief Find diagrams by tags
     * 
     * @param tags The tags to search for
     * @param match_all Whether all tags must match (AND) or any tag (OR)
     * @return std::vector<DiagramMetadata> Matching diagram metadata
     */
    virtual std::vector<DiagramMetadata> findDiagramsByTags(
        const std::vector<std::string>& tags,
        bool match_all = true) = 0;
    
    /**
     * @brief Find vectors by tags
     * 
     * @param tags The tags to search for
     * @param match_all Whether all tags must match (AND) or any tag (OR)
     * @return std::vector<VectorMetadata> Matching vector metadata
     */
    virtual std::vector<VectorMetadata> findVectorsByTags(
        const std::vector<std::string>& tags,
        bool match_all = true) = 0;
    
    /**
     * @brief Find vectors for a diagram
     * 
     * @param diagram_id The ID of the diagram
     * @return std::vector<VectorMetadata> Metadata for vectors derived from the diagram
     */
    virtual std::vector<VectorMetadata> findVectorsForDiagram(const std::string& diagram_id) = 0;
    
    /**
     * @brief Vectorize and store a diagram
     * 
     * @param diagram_id The ID of the diagram to vectorize
     * @param vectorizer The vectorizer to use
     * @param tags Optional tags for the vector
     * @return std::string The ID of the stored vector
     */
    virtual std::string vectorizeAndStore(
        const std::string& diagram_id,
        const tda::vectorization::Vectorizer& vectorizer,
        const std::vector<std::string>& tags = {}) = 0;
    
    /**
     * @brief Asynchronously vectorize and store a diagram
     * 
     * @param diagram_id The ID of the diagram to vectorize
     * @param vectorizer The vectorizer to use
     * @param tags Optional tags for the vector
     * @return std::future<std::string> Future for the ID of the stored vector
     */
    virtual std::future<std::string> vectorizeAndStoreAsync(
        const std::string& diagram_id,
        std::shared_ptr<tda::vectorization::Vectorizer> vectorizer,
        const std::vector<std::string>& tags = {}) = 0;
    
    /**
     * @brief Delete a diagram and all its vectors
     * 
     * @param id The ID of the diagram to delete
     * @return bool True if deleted, false if not found
     */
    virtual bool deleteDiagram(const std::string& id) = 0;
    
    /**
     * @brief Delete a vector
     * 
     * @param id The ID of the vector to delete
     * @return bool True if deleted, false if not found
     */
    virtual bool deleteVector(const std::string& id) = 0;
    
    /**
     * @brief Add tags to a diagram
     * 
     * @param id The ID of the diagram
     * @param tags The tags to add
     * @return bool True if successful, false if diagram not found
     */
    virtual bool addTagsToDiagram(const std::string& id, const std::vector<std::string>& tags) = 0;
    
    /**
     * @brief Add tags to a vector
     * 
     * @param id The ID of the vector
     * @param tags The tags to add
     * @return bool True if successful, false if vector not found
     */
    virtual bool addTagsToVector(const std::string& id, const std::vector<std::string>& tags) = 0;
    
    /**
     * @brief Remove tags from a diagram
     * 
     * @param id The ID of the diagram
     * @param tags The tags to remove
     * @return bool True if successful, false if diagram not found
     */
    virtual bool removeTagsFromDiagram(const std::string& id, const std::vector<std::string>& tags) = 0;
    
    /**
     * @brief Remove tags from a vector
     * 
     * @param id The ID of the vector
     * @param tags The tags to remove
     * @return bool True if successful, false if vector not found
     */
    virtual bool removeTagsFromVector(const std::string& id, const std::vector<std::string>& tags) = 0;
};

} // namespace storage
} // namespace tda
