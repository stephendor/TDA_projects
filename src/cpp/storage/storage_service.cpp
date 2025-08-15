#include <tda/storage/storage_service.hpp>
#include <tda/storage/postgresql_storage.hpp>
#include <tda/storage/mongodb_storage.hpp>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <nlohmann/json.hpp>

namespace tda {
namespace storage {

std::unique_ptr<StorageService> StorageServiceFactory::create(const std::string& type, const nlohmann::json& config) {
    if (type == "postgresql" || type == "postgres") {
        try {
            PostgreSQLStorage::ConnectionConfig db_config;
            db_config.host = config.value("host", "localhost");
            db_config.port = config.value("port", 5432);
            db_config.database = config.at("database");
            db_config.user = config.value("user", "postgres");
            db_config.password = config.value("password", "");
            db_config.schema = config.value("schema", "public");
            
            return std::make_unique<PostgreSQLStorage>(db_config);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create PostgreSQL storage: " + std::string(e.what()));
        }
    } else if (type == "mongodb" || type == "mongo") {
        try {
            MongoDBStorage::ConnectionConfig db_config;
            db_config.host = config.value("host", "localhost");
            db_config.port = config.value("port", 27017);
            db_config.database = config.at("database");
            db_config.user = config.value("user", "");
            db_config.password = config.value("password", "");
            
            return std::make_unique<MongoDBStorage>(db_config);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create MongoDB storage: " + std::string(e.what()));
        }
    } else {
        throw std::runtime_error("Unknown storage type: " + type);
    }
}

std::unique_ptr<StorageService> StorageServiceFactory::createFromJSON(const std::string& json_str) {
    try {
        nlohmann::json json = nlohmann::json::parse(json_str);
        std::string type = json.at("type");
        nlohmann::json config = json.at("config");
        
        return create(type, config);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create storage from JSON: " + std::string(e.what()));
    }
}

HybridStorageService::HybridStorageService(
    std::unique_ptr<StorageService> metadata_storage,
    std::unique_ptr<StorageService> data_storage)
    : metadata_storage_(std::move(metadata_storage)),
      data_storage_(std::move(data_storage)) {}

bool HybridStorageService::storeDiagram(
    const std::string& id,
    const StorageService::PersistenceDiagram& diagram,
    const DiagramMetadata& metadata) {
    
    // Store metadata in the metadata storage (e.g., PostgreSQL)
    bool metadata_success = metadata_storage_->storeDiagram(id, diagram, metadata);
    
    if (!metadata_success) {
        std::cerr << "Failed to store diagram metadata" << std::endl;
        return false;
    }
    
    // Store diagram data in the data storage (e.g., MongoDB)
    bool data_success = data_storage_->storeDiagram(id, diagram, metadata);
    
    if (!data_success) {
        std::cerr << "Failed to store diagram data" << std::endl;
        // Consider rolling back metadata storage here in a real implementation
        return false;
    }
    
    return true;
}

bool HybridStorageService::storeVectorization(
    const std::string& diagram_id,
    const std::string& vectorizer_id,
    const StorageService::FeatureVector& vector,
    const VectorMetadata& metadata) {
    
    // Store metadata in the metadata storage
    bool metadata_success = metadata_storage_->storeVectorization(diagram_id, vectorizer_id, vector, metadata);
    
    if (!metadata_success) {
        std::cerr << "Failed to store vectorization metadata" << std::endl;
        return false;
    }
    
    // Store vector data in the data storage
    bool data_success = data_storage_->storeVectorization(diagram_id, vectorizer_id, vector, metadata);
    
    if (!data_success) {
        std::cerr << "Failed to store vectorization data" << std::endl;
        // Consider rolling back metadata storage here in a real implementation
        return false;
    }
    
    return true;
}

std::optional<StorageService::PersistenceDiagram> HybridStorageService::getDiagram(const std::string& id) {
    // Try to get diagram from data storage first, as it contains the full data
    auto diagram = data_storage_->getDiagram(id);
    
    if (diagram) {
        return diagram;
    }
    
    // Fall back to metadata storage
    return metadata_storage_->getDiagram(id);
}

std::optional<StorageService::FeatureVector> HybridStorageService::getVectorization(
    const std::string& diagram_id,
    const std::string& vectorizer_id) {
    
    // Try to get vector from data storage first
    auto vector = data_storage_->getVectorization(diagram_id, vectorizer_id);
    
    if (vector) {
        return vector;
    }
    
    // Fall back to metadata storage
    return metadata_storage_->getVectorization(diagram_id, vectorizer_id);
}

std::vector<std::string> HybridStorageService::listDiagrams(const QueryFilter& filter) {
    // Use metadata storage for listing, as it should have all diagram IDs and efficient querying
    return metadata_storage_->listDiagrams(filter);
}

bool HybridStorageService::deleteDiagram(const std::string& id) {
    // Delete from both storages
    bool metadata_success = metadata_storage_->deleteDiagram(id);
    bool data_success = data_storage_->deleteDiagram(id);
    
    // Return true only if both succeeded
    return metadata_success && data_success;
}

std::unique_ptr<StorageService> createHybridStorageService(
    const std::string& metadata_type,
    const nlohmann::json& metadata_config,
    const std::string& data_type,
    const nlohmann::json& data_config) {
    
    try {
        auto metadata_storage = StorageServiceFactory::create(metadata_type, metadata_config);
        auto data_storage = StorageServiceFactory::create(data_type, data_config);
        
        return std::make_unique<HybridStorageService>(std::move(metadata_storage), std::move(data_storage));
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create hybrid storage service: " + std::string(e.what()));
    }
}

} // namespace storage
} // namespace tda
