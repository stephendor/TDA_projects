#pragma once

#include <tda/core/types.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <future>
#include <nlohmann/json.hpp>

namespace tda {
namespace storage {

// Basic metadata structures used by current storage implementations
struct DiagramMetadata {
    std::string pointcloud_id;
    std::string algorithm;
    nlohmann::json parameters;
    std::time_t timestamp{};
    std::vector<int> dimensions;
};

struct VectorMetadata {
    nlohmann::json vectorizer_config;
    size_t dimension{};
    std::time_t timestamp{};
};

struct QueryFilter {
    std::string pointcloud_id;
    std::string algorithm;
    std::time_t min_timestamp{};
    std::time_t max_timestamp{};
};

/**
 * @brief Abstract storage service interface (current example implementation)
 */
class StorageService {
public:
    using PersistenceDiagram = std::vector<tda::core::PersistencePair>;
    using FeatureVector = std::vector<double>;

    virtual ~StorageService() = default;

    virtual bool storeDiagram(
        const std::string& id,
        const PersistenceDiagram& diagram,
        const DiagramMetadata& metadata) = 0;

    virtual bool storeVectorization(
        const std::string& diagram_id,
        const std::string& vectorizer_id,
        const FeatureVector& vector,
        const VectorMetadata& metadata) = 0;

    virtual std::optional<PersistenceDiagram> getDiagram(const std::string& id) = 0;

    virtual std::optional<FeatureVector> getVectorization(
        const std::string& diagram_id,
        const std::string& vectorizer_id) = 0;

    virtual std::vector<std::string> listDiagrams(const QueryFilter& filter) = 0;

    virtual bool deleteDiagram(const std::string& id) = 0;
};

// Forward declarations
class StorageServiceFactory;
class HybridStorageService;

// Hybrid storage combines a metadata store and a data store
class HybridStorageService : public StorageService {
public:
    HybridStorageService(
        std::unique_ptr<StorageService> metadata_storage,
        std::unique_ptr<StorageService> data_storage);

    // StorageService interface
    bool storeDiagram(
        const std::string& id,
        const StorageService::PersistenceDiagram& diagram,
        const DiagramMetadata& metadata) override;

    bool storeVectorization(
        const std::string& diagram_id,
        const std::string& vectorizer_id,
        const StorageService::FeatureVector& vector,
        const VectorMetadata& metadata) override;

    std::optional<StorageService::PersistenceDiagram> getDiagram(const std::string& id) override;

    std::optional<StorageService::FeatureVector> getVectorization(
        const std::string& diagram_id,
        const std::string& vectorizer_id) override;

    std::vector<std::string> listDiagrams(const QueryFilter& filter) override;

    bool deleteDiagram(const std::string& id) override;

private:
    std::unique_ptr<StorageService> metadata_storage_;
    std::unique_ptr<StorageService> data_storage_;
};

class StorageServiceFactory {
public:
    static std::unique_ptr<StorageService> create(const std::string& type, const nlohmann::json& config);
    static std::unique_ptr<StorageService> createFromJSON(const std::string& json_str);
};

// Convenience factory for hybrid storage composed of a metadata and a data backend
std::unique_ptr<StorageService> createHybridStorageService(
    const std::string& metadata_type,
    const nlohmann::json& metadata_config,
    const std::string& data_type,
    const nlohmann::json& data_config);

} // namespace storage
} // namespace tda
